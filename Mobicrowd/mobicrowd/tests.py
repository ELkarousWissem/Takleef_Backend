from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from mobicrowd.models.Users import (
    Organization,
    OrganizationInvitation,
    OrganizationLicenceKey,
    OrganizationMembership,
    Requester,
    User,
)


class OrganizationAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.admin = self._user("admin@example.com", role="Admin")
        self.non_admin = self._user("worker@example.com", role="Worker")
        self.rep_user = self._user(
            "rep@example.com",
            role="Requester",
            is_requester=True,
            is_representative=True,
        )
        self.rep = self._requester(self.rep_user, org_name="Acme")
        self.org = self._organization(self.rep, name="Acme", email="acme@example.com", status="active")

    def _user(self, email, **extra):
        defaults = {
            "password": "pass12345",
            "fullName": email.split("@")[0],
            "mobile_phone": "12345678",
            "role": "Worker",
            "is_active": True,
        }
        defaults.update(extra)
        return User.objects.create_user(email=email, **defaults)

    def _requester(self, user, org_name="", location="Tunis", approved=True):
        return Requester.objects.create(
            user=user,
            approved=approved,
            organization_name=org_name,
            location=location,
        )

    def _organization(
        self,
        requester,
        *,
        name,
        email,
        status="active",
        licence_requester=2,
        licence_contributor=2,
    ):
        return Organization.objects.create(
            representative=requester,
            name=name,
            email=email,
            status=status,
            licence_requester=licence_requester,
            licence_contributor=licence_contributor,
        )

    def _member(self, org, user, role="contributor", status="active"):
        licence_key = OrganizationLicenceKey.objects.filter(
            organization=org,
            role=role,
            status="idle",
        ).first()
        self.assertIsNotNone(licence_key, f"No idle {role} licence available")
        membership = OrganizationMembership.objects.create(
            organization=org,
            user=user,
            role=role,
            licence_key=licence_key,
            status=status,
        )
        licence_key.activate_for_user(user)
        return membership

    def _invitation(self, org, email, role="contributor", status="sent", expires_at=None):
        licence_key = OrganizationLicenceKey.objects.filter(
            organization=org,
            role=role,
            status="idle",
        ).first()
        self.assertIsNotNone(licence_key, f"No idle {role} licence available")
        invitation = OrganizationInvitation.objects.create(
            organization=org,
            licence_key=licence_key,
            email=email,
            role=role,
            status=status,
            expires_at=expires_at or timezone.now() + timedelta(hours=24),
        )
        if status == "sent":
            licence_key.mark_as_sent(email)
        return invitation

    def _authenticate(self, user):
        self.client.force_authenticate(user=user)

    def _logout(self):
        self.client.force_authenticate(user=None)

    def _patch_org_emails(self):
        return patch.multiple(
            "mobicrowd.serializers.organizationSerializer",
            build_reset_password_link=lambda user: "https://frontend/reset",
            send_forget_password_email=lambda **kwargs: None,
            send_new_representative_welcome_email=lambda **kwargs: None,
            send_old_representative_removed_email=lambda **kwargs: None,
        )

    # 1. Administration des organisations

    def test_admin_can_create_organization_with_representative(self):
        self._authenticate(self.admin)
        with self._patch_org_emails():
            response = self.client.post(
                "/api/organizations/create/",
                {
                    "name": "Created Org",
                    "email": "created@example.com",
                    "representative_email": "created-rep@example.com",
                    "representative_full_name": "Created Rep",
                    "representative_mobile_phone": "11111111",
                    "licence_requester": 1,
                    "licence_contributor": 1,
                },
                format="json",
            )

        self.assertEqual(response.status_code, 201)
        self.assertTrue(Organization.objects.filter(email="created@example.com").exists())
        self.assertTrue(User.objects.filter(email="created-rep@example.com", is_representative=True).exists())

    def test_non_admin_cannot_create_or_list_organizations(self):
        self._authenticate(self.non_admin)

        create_response = self.client.post(
            "/api/organizations/create/",
            {"name": "Nope", "email": "nope@example.com", "representative_email": "x@example.com"},
            format="json",
        )
        list_response = self.client.get("/api/organizations/")

        self.assertEqual(create_response.status_code, 403)
        self.assertEqual(list_response.status_code, 403)

    def test_admin_can_list_update_and_delete_organization(self):
        self._authenticate(self.admin)
        new_rep_user = self._user(
            "new-rep@example.com",
            role="Requester",
            is_requester=True,
            is_representative=False,
        )
        self._requester(new_rep_user, org_name="", approved=True)

        list_response = self.client.get("/api/organizations/")
        self.assertEqual(list_response.status_code, 200)
        self.assertGreaterEqual(len(list_response.data), 1)

        with self._patch_org_emails():
            update_response = self.client.put(
                f"/api/organizations/update/{self.org.id}/",
                {
                    "name": "Updated Acme",
                    "email": "updated-acme@example.com",
                    "licence_requester": 3,
                    "licence_contributor": 4,
                    "representative_email": new_rep_user.email,
                },
                format="json",
            )
        self.assertEqual(update_response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(self.org.name, "Updated Acme")
        self.assertEqual(self.org.email, "updated-acme@example.com")
        self.assertEqual(self.org.licence_requester, 3)
        self.assertEqual(self.org.licence_contributor, 4)
        self.assertEqual(self.org.representative.user.email, new_rep_user.email)

        delete_response = self.client.delete(f"/api/organizations/delete/{self.org.id}/")
        self.assertEqual(delete_response.status_code, 204)
        self.assertFalse(Organization.objects.filter(id=self.org.id).exists())

    def test_update_delete_missing_organization_returns_404(self):
        self._authenticate(self.admin)
        self.assertEqual(
            self.client.put("/api/organizations/update/999999/", {"name": "Missing"}, format="json").status_code,
            404,
        )
        self.assertEqual(self.client.delete("/api/organizations/delete/999999/").status_code, 404)

    # 2. Vérification représentant

    def test_representative_email_check_cases(self):
        self._authenticate(self.admin)
        active_requester_user = self._user(
            "active-requester@example.com",
            role="Requester",
            is_requester=True,
        )
        self._requester(active_requester_user)
        non_requester_user = self._user("plain-worker@example.com", role="Worker", is_requester=False)

        missing = self.client.get("/api/representative-email-check/", {"email": "missing@example.com"})
        active = self.client.get("/api/representative-email-check/", {"email": active_requester_user.email})
        non_requester = self.client.get("/api/representative-email-check/", {"email": non_requester_user.email})
        already_rep = self.client.get("/api/representative-email-check/", {"email": self.rep_user.email})

        self.assertEqual(missing.status_code, 200)
        self.assertFalse(missing.data["exists_in_user"])
        self.assertTrue(active.data["exists_in_user"])
        self.assertTrue(active.data["is_requester"])
        self.assertTrue(active.data["requester_is_activated"])
        self.assertTrue(non_requester.data["exists_in_user"])
        self.assertFalse(non_requester.data["is_requester"])
        self.assertTrue(already_rep.data["already_representative_elsewhere"])

        self._authenticate(self.non_admin)
        forbidden = self.client.get("/api/representative-email-check/", {"email": self.rep_user.email})
        self.assertEqual(forbidden.status_code, 403)

    def test_role_representative_check_cases(self):
        pending_rep_user = self._user(
            "pending-role-rep@example.com",
            role="Requester",
            is_requester=True,
            is_representative=True,
        )
        pending_rep = self._requester(pending_rep_user, org_name="Pending")
        self._organization(pending_rep, name="Pending", email="pending-role@example.com", status="pending")

        self._authenticate(pending_rep_user)
        self.assertTrue(self.client.get("/api/auth/rep/").data["can_activate_organization"])

        requester_user = self._user("simple-requester@example.com", role="Requester", is_requester=True)
        self._requester(requester_user)
        self._authenticate(requester_user)
        self.assertFalse(self.client.get("/api/auth/rep/").data["can_activate_organization"])

        self._authenticate(self.non_admin)
        self.assertFalse(self.client.get("/api/auth/rep/").data["can_activate_organization"])

        self._authenticate(self.admin)
        self.assertFalse(self.client.get("/api/auth/rep/").data["can_activate_organization"])

    # 3. Organisation du représentant

    def test_my_representative_organization_cases(self):
        self._authenticate(self.rep_user)
        success = self.client.get("/api/organizations/my-representative-organization/")
        self.assertEqual(success.status_code, 200)
        self.assertEqual(success.data["id"], self.org.id)

        self._authenticate(self.non_admin)
        self.assertEqual(self.client.get("/api/organizations/my-representative-organization/").status_code, 403)

        rep_without_profile = self._user(
            "no-profile-rep@example.com",
            role="Requester",
            is_requester=True,
            is_representative=True,
        )
        self._authenticate(rep_without_profile)
        self.assertEqual(self.client.get("/api/organizations/my-representative-organization/").status_code, 404)

        rep_no_org_user = self._user(
            "no-org-rep@example.com",
            role="Requester",
            is_requester=True,
            is_representative=True,
        )
        self._requester(rep_no_org_user)
        self._authenticate(rep_no_org_user)
        self.assertEqual(self.client.get("/api/organizations/my-representative-organization/").status_code, 404)

    # 4. Activation organisation

    def test_activate_organization_cases(self):
        pending_rep_user = self._user(
            "activate-rep@example.com",
            role="Requester",
            is_requester=True,
            is_representative=True,
        )
        pending_rep = self._requester(pending_rep_user, org_name="Activate Org")
        pending_org = self._organization(
            pending_rep,
            name="Activate Org",
            email="activate@example.com",
            status="pending",
        )

        self._authenticate(self.non_admin)
        forbidden = self.client.put(
            f"/api/organizations/activate/{pending_org.id}/",
            {"activation_code": pending_org.activation_code},
            format="json",
        )
        self.assertEqual(forbidden.status_code, 403)

        self._authenticate(pending_rep_user)
        missing_code = self.client.put(f"/api/organizations/activate/{pending_org.id}/", {}, format="json")
        wrong_code = self.client.put(
            f"/api/organizations/activate/{pending_org.id}/",
            {"activation_code": "WRONG"},
            format="json",
        )
        success = self.client.put(
            f"/api/organizations/activate/{pending_org.id}/",
            {"activation_code": pending_org.activation_code},
            format="json",
        )
        already_active = self.client.put(
            f"/api/organizations/activate/{pending_org.id}/",
            {"activation_code": pending_org.activation_code},
            format="json",
        )
        missing_org = self.client.put(
            "/api/organizations/activate/999999/",
            {"activation_code": "ANY"},
            format="json",
        )

        self.assertEqual(missing_code.status_code, 400)
        self.assertEqual(wrong_code.status_code, 400)
        self.assertEqual(success.status_code, 200)
        self.assertEqual(already_active.status_code, 400)
        self.assertEqual(missing_org.status_code, 404)

    # 5. Mes organisations

    def test_my_organizations_cases(self):
        requester_user = self._user("org-requester@example.com", role="Requester", is_requester=True)
        contributor_user = self._user("org-contributor@example.com", role="Worker")
        empty_user = self._user("empty@example.com", role="Worker")
        self._member(self.org, requester_user, role="requester")
        self._member(self.org, contributor_user, role="contributor")

        self._authenticate(self.rep_user)
        rep_response = self.client.get("/api/organizations/my-organizations/")
        self.assertEqual(rep_response.status_code, 200)
        self.assertEqual(rep_response.data[0]["myRole"], "Requester")
        self.assertTrue(rep_response.data[0]["isRepresentative"])

        self._authenticate(requester_user)
        requester_response = self.client.get("/api/organizations/my-organizations/")
        self.assertEqual(requester_response.data[0]["myRole"], "Requester")

        self._authenticate(contributor_user)
        contributor_response = self.client.get("/api/organizations/my-organizations/")
        self.assertEqual(contributor_response.data[0]["myRole"], "Contributor")

        self._authenticate(empty_user)
        self.assertEqual(self.client.get("/api/organizations/my-organizations/").data, [])

    # 6. Détail organisation

    def test_organization_detail_cases(self):
        member_user = self._user("detail-member@example.com", role="Worker")
        non_member = self._user("detail-outsider@example.com", role="Worker")
        self._member(self.org, member_user, role="contributor")

        self._authenticate(self.rep_user)
        self.assertEqual(self.client.get(f"/api/organizations/{self.org.id}/").status_code, 200)

        self._authenticate(member_user)
        self.assertEqual(self.client.get(f"/api/organizations/{self.org.id}/").status_code, 200)

        self._authenticate(non_member)
        self.assertEqual(self.client.get(f"/api/organizations/{self.org.id}/").status_code, 403)

        self.org.status = "pending"
        self.org.save(update_fields=["status"])
        self._authenticate(self.rep_user)
        self.assertEqual(self.client.get(f"/api/organizations/{self.org.id}/").status_code, 403)
        self.assertEqual(self.client.get("/api/organizations/999999/").status_code, 404)

    # 7. Membres

    def test_members_list_cases_and_memberships_route_disabled(self):
        member_user = self._user("member-list@example.com", role="Worker")
        outsider = self._user("member-outsider@example.com", role="Worker")
        self._member(self.org, member_user, role="contributor")

        self._authenticate(self.rep_user)
        rep_response = self.client.get(f"/api/organizations/{self.org.id}/members/")
        self.assertEqual(rep_response.status_code, 200)
        self.assertGreaterEqual(len(rep_response.data), 1)

        self._authenticate(member_user)
        self.assertEqual(self.client.get(f"/api/organizations/{self.org.id}/members/").status_code, 403)

        self._authenticate(outsider)
        self.assertEqual(self.client.get(f"/api/organizations/{self.org.id}/members/").status_code, 403)

        self._authenticate(self.rep_user)
        disabled = self.client.post(
            f"/api/organizations/{self.org.id}/memberships/",
            {"email": outsider.email, "role": "contributor"},
            format="json",
        )
        self.assertEqual(disabled.status_code, 410)

    # 8. Licences

    def test_license_summary_cases(self):
        requester_member = self._user("license-requester@example.com", role="Requester", is_requester=True)
        contributor_member = self._user("license-contributor@example.com", role="Worker")
        outsider = self._user("license-outsider@example.com", role="Worker")
        self._member(self.org, requester_member, role="requester")
        self._member(self.org, contributor_member, role="contributor")

        self._authenticate(self.rep_user)
        rep_response = self.client.get(f"/api/organizations/{self.org.id}/license-summary/")
        self.assertEqual(rep_response.status_code, 200)
        self.assertEqual(rep_response.data["requesterTotal"], self.org.licence_requester)
        self.assertEqual(rep_response.data["requesterUsed"], 1)
        self.assertEqual(rep_response.data["contributorUsed"], 1)

        self._authenticate(contributor_member)
        self.assertEqual(self.client.get(f"/api/organizations/{self.org.id}/license-summary/").status_code, 200)

        self._authenticate(outsider)
        self.assertEqual(self.client.get(f"/api/organizations/{self.org.id}/license-summary/").status_code, 403)

    # 9. Invitations

    @patch("mobicrowd.views.apis.organization.send_organization_invitation_email")
    def test_invite_members_cases(self, send_email):
        self._authenticate(self.rep_user)
        success = self.client.post(
            f"/api/organizations/{self.org.id}/invite/",
            {"emails": ["invite1@example.com"], "role": "contributor"},
            format="json",
        )
        self.assertEqual(success.status_code, 200)
        self.assertEqual(len(success.data["created"]), 1)
        self.assertEqual(send_email.call_count, 1)

        duplicate = self.client.post(
            f"/api/organizations/{self.org.id}/invite/",
            {"emails": ["invite1@example.com"], "role": "contributor"},
            format="json",
        )
        self.assertEqual(duplicate.data["errors"][0]["error"], "Pending invitation already exists.")

        member_user = self._user("already-member@example.com", role="Worker")
        self._member(self.org, member_user, role="contributor")
        already_member = self.client.post(
            f"/api/organizations/{self.org.id}/invite/",
            {"emails": [member_user.email], "role": "contributor"},
            format="json",
        )
        self.assertEqual(already_member.data["errors"][0]["error"], "This user already belongs to this organization.")

        no_license = self.client.post(
            f"/api/organizations/{self.org.id}/invite/",
            {"emails": ["no-license@example.com"], "role": "contributor"},
            format="json",
        )
        self.assertEqual(no_license.data["errors"][0]["error"], "No available contributor licence.")

        inactive_rep_user = self._user(
            "inactive-rep@example.com",
            role="Requester",
            is_requester=True,
            is_representative=True,
        )
        inactive_rep = self._requester(inactive_rep_user, org_name="Inactive")
        inactive_org = self._organization(inactive_rep, name="Inactive", email="inactive@example.com", status="pending")
        self._authenticate(inactive_rep_user)
        inactive = self.client.post(
            f"/api/organizations/{inactive_org.id}/invite/",
            {"emails": ["inactive@example.com"], "role": "requester"},
            format="json",
        )
        self.assertEqual(inactive.status_code, 400)

        self._authenticate(self.non_admin)
        forbidden = self.client.post(
            f"/api/organizations/{self.org.id}/invite/",
            {"emails": ["forbidden@example.com"], "role": "requester"},
            format="json",
        )
        self.assertEqual(forbidden.status_code, 403)

    def test_invitations_list_and_cancel_cases(self):
        invitation = self._invitation(self.org, "list@example.com", role="contributor")

        self._authenticate(self.rep_user)
        list_response = self.client.get(f"/api/organizations/{self.org.id}/invitations/")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.data[0]["email"], invitation.email)

        cancel_response = self.client.delete(f"/api/organization-invitations/{invitation.id}/cancel/")
        self.assertEqual(cancel_response.status_code, 200)
        invitation.refresh_from_db()
        self.assertEqual(invitation.status, "cancelled")

        second_invitation = self._invitation(self.org, "second-list@example.com", role="requester")
        self._authenticate(self.non_admin)
        forbidden = self.client.delete(f"/api/organization-invitations/{second_invitation.id}/cancel/")
        self.assertEqual(forbidden.status_code, 403)

        self._authenticate(self.rep_user)
        second_invitation.status = "accepted"
        second_invitation.save(update_fields=["status"])
        invalid_cancel = self.client.delete(f"/api/organization-invitations/{second_invitation.id}/cancel/")
        self.assertEqual(invalid_cancel.status_code, 400)

    # 10. Acceptation / refus invitation

    def test_invitation_detail_cases(self):
        invitation = self._invitation(self.org, "detail-invite@example.com", role="requester")

        valid = self.client.get("/api/organization-invitations/detail/", {"token": invitation.token})
        missing = self.client.get("/api/organization-invitations/detail/")
        invalid = self.client.get("/api/organization-invitations/detail/", {"token": "invalid-token"})

        self.assertEqual(valid.status_code, 200)
        self.assertNotIn("email", valid.data)
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(invalid.status_code, 404)

        expired_invitation = self._invitation(
            self.org,
            "expired@example.com",
            role="requester",
            expires_at=timezone.now() - timedelta(hours=1),
        )
        expired = self.client.get("/api/organization-invitations/detail/", {"token": expired_invitation.token})
        expired_invitation.refresh_from_db()
        self.assertEqual(expired.status_code, 200)
        self.assertEqual(expired_invitation.status, "expired")
        self.assertTrue(expired.data["isExpired"])

    def test_accept_invitation_cases(self):
        invited_user = self._user("accept@example.com", role="Worker")
        invitation = self._invitation(self.org, invited_user.email, role="contributor")
        self._authenticate(invited_user)

        missing = self.client.post("/api/organizations/invitations/accept/", {}, format="json")
        self.assertEqual(missing.status_code, 400)

        wrong_user = self._user("wrong-accept@example.com", role="Worker")
        self._authenticate(wrong_user)
        forbidden = self.client.post(
            "/api/organizations/invitations/accept/",
            {"token": invitation.token},
            format="json",
        )
        self.assertEqual(forbidden.status_code, 403)

        self._authenticate(invited_user)
        success = self.client.post(
            "/api/organizations/invitations/accept/",
            {"token": invitation.token},
            format="json",
        )
        self.assertEqual(success.status_code, 200)
        self.assertTrue(OrganizationMembership.objects.filter(organization=self.org, user=invited_user).exists())

        reused = self.client.post(
            "/api/organizations/invitations/accept/",
            {"token": invitation.token},
            format="json",
        )
        self.assertEqual(reused.status_code, 400)

        expired_user = self._user("expired-accept@example.com", role="Worker")
        expired_invitation = self._invitation(
            self.org,
            expired_user.email,
            role="contributor",
            expires_at=timezone.now() - timedelta(hours=1),
        )
        self._authenticate(expired_user)
        expired = self.client.post(
            "/api/organizations/invitations/accept/",
            {"token": expired_invitation.token},
            format="json",
        )
        self.assertEqual(expired.status_code, 400)

    def test_decline_invitation_cases(self):
        invited_user = self._user("decline@example.com", role="Worker")
        invitation = self._invitation(self.org, invited_user.email, role="contributor")

        self._authenticate(invited_user)
        missing = self.client.post("/api/organization-invitations/decline/", {}, format="json")
        self.assertEqual(missing.status_code, 400)

        wrong_user = self._user("wrong-decline@example.com", role="Worker")
        self._authenticate(wrong_user)
        forbidden = self.client.post(
            "/api/organization-invitations/decline/",
            {"token": invitation.token},
            format="json",
        )
        self.assertEqual(forbidden.status_code, 403)

        self._authenticate(invited_user)
        success = self.client.post(
            "/api/organization-invitations/decline/",
            {"token": invitation.token},
            format="json",
        )
        self.assertEqual(success.status_code, 200)
        invitation.refresh_from_db()
        self.assertEqual(invitation.status, "declined")

        reused = self.client.post(
            "/api/organization-invitations/decline/",
            {"token": invitation.token},
            format="json",
        )
        self.assertEqual(reused.status_code, 400)

    # Security regressions added during this pass

    def test_superuser_can_access_admin_organization_events(self):
        superuser = self._user(
            "superuser@example.com",
            role="Requester",
            is_requester=True,
            is_superuser=True,
        )
        self._authenticate(superuser)

        response = self.client.get("/api/organizations/events/")

        self.assertEqual(response.status_code, 200)

    def test_organization_wide_event_listing_is_disabled(self):
        contributor = self._user("event-list-contributor@example.com", role="Worker")
        self._member(self.org, contributor, role="contributor")
        self._authenticate(contributor)

        response = self.client.get(f"/api/organizations/{self.org.id}/events/")

        self.assertEqual(response.status_code, 410)
